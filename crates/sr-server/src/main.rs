//! Star Realms server: serves the web client and card art, and hosts games.
//!
//! Identity is a random token kept in the browser (no accounts). A game is
//! created against a bot or with an open seat; the invite link fills the open
//! seat, and anyone else with the link spectates. Every game is stored as a
//! replay in SQLite, so restarts lose nothing.

mod db;
mod rooms;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use db::{Db, Player};
use futures_util::{SinkExt, StreamExt};
use rooms::{random_id, App, Move, Room, Seat, ServerMsg};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::set_header::SetResponseHeaderLayer;

#[derive(Parser)]
#[command(about = "Star Realms game server")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:3030", env = "SR_BIND")]
    bind: String,
    /// SQLite database file.
    #[arg(long, default_value = "starrealms.db", env = "SR_DB")]
    db: PathBuf,
    /// Built web client (`npm run build` in web/).
    #[arg(long, default_value = "web/dist", env = "SR_WEB")]
    web: PathBuf,
    /// Card art fetched by tools/fetch_assets.py.
    #[arg(long, default_value = "assets", env = "SR_ASSETS")]
    assets: PathBuf,
    /// Directory of exported `.srnn` models offered as opponents.
    #[arg(long, default_value = "models", env = "SR_MODELS")]
    models: PathBuf,
    /// Pause between bot actions, so humans can follow the bot's turn.
    #[arg(long, default_value_t = 450, env = "SR_BOT_DELAY_MS")]
    bot_delay_ms: u64,
}

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

fn internal(e: impl std::fmt::Display) -> ApiError {
    tracing::error!("{e}");
    ApiError(StatusCode::INTERNAL_SERVER_ERROR, "internal error".into())
}

type ApiResult<T> = Result<T, ApiError>;

fn clean_name(name: &str) -> ApiResult<String> {
    let n: String = name.trim().chars().filter(|c| !c.is_control()).take(24).collect();
    if n.is_empty() {
        return Err(bad("pick a display name"));
    }
    Ok(n)
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers.get(header::AUTHORIZATION)?.to_str().ok()?.strip_prefix("Bearer ")
}

fn player(app: &App, token: Option<&str>) -> ApiResult<Option<Player>> {
    match token {
        Some(t) => app.db.player_by_token(t).map_err(internal),
        None => Ok(None),
    }
}

fn require_player(app: &App, headers: &HeaderMap) -> ApiResult<Player> {
    player(app, bearer(headers))?.ok_or(ApiError(StatusCode::UNAUTHORIZED, "unknown player; set a name first".into()))
}

fn find_room(app: &App, id: &str) -> ApiResult<Arc<Room>> {
    app.room(id).map_err(internal)?.ok_or(ApiError(StatusCode::NOT_FOUND, "no such game".into()))
}

#[derive(Deserialize)]
struct NameBody {
    name: String,
}

async fn create_player(State(app): State<Arc<App>>, Json(b): Json<NameBody>) -> ApiResult<Json<serde_json::Value>> {
    let name = clean_name(&b.name)?;
    let (id, token) = (random_id(12), random_id(32));
    app.db.create_player(&id, &token, &name, rooms::now()).map_err(internal)?;
    Ok(Json(json!({ "id": id, "token": token, "name": name })))
}

async fn me(State(app): State<Arc<App>>, headers: HeaderMap) -> ApiResult<Json<serde_json::Value>> {
    let p = require_player(&app, &headers)?;
    Ok(Json(json!({ "id": p.id, "name": p.name })))
}

async fn rename(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(b): Json<NameBody>,
) -> ApiResult<Json<serde_json::Value>> {
    let p = require_player(&app, &headers)?;
    let name = clean_name(&b.name)?;
    app.db.rename_player(&p.id, &name).map_err(internal)?;
    Ok(Json(json!({ "id": p.id, "name": name })))
}

async fn catalog(State(app): State<Arc<App>>) -> Response {
    ([(header::CONTENT_TYPE, "application/json")], app.catalog.clone()).into_response()
}

/// Strongest first: the heuristic bot, trained models, then the weak bots.
async fn list_opponents(State(app): State<Arc<App>>) -> Json<serde_json::Value> {
    let rank = |spec: &str| match spec {
        "heuristic" => 0,
        "greedy" => 2,
        "random" => 3,
        _ => 1,
    };
    let mut list: Vec<_> = app.opponents.values().collect();
    list.sort_by_key(|o| rank(&o.spec));
    Json(json!(list.iter().map(|o| json!({ "spec": o.spec, "name": o.name })).collect::<Vec<_>>()))
}

#[derive(Deserialize)]
struct NewGame {
    sets: Vec<String>,
    /// "human" for an invite link, else an opponent spec from /api/opponents.
    opponent: String,
    /// "first" | "second" | "random"
    #[serde(default)]
    seat: Option<String>,
}

async fn create_game(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(b): Json<NewGame>,
) -> ApiResult<Json<serde_json::Value>> {
    let p = require_player(&app, &headers)?;
    const SETS: [&str; 3] = ["core", "colony_wars", "frontiers"];
    let mut sets: Vec<String> = SETS.iter().filter(|s| b.sets.iter().any(|x| x == *s)).map(|s| s.to_string()).collect();
    sets.dedup();
    if sets.is_empty() || sets.len() != b.sets.len() {
        return Err(bad("choose sets from core, colony_wars, frontiers"));
    }
    let other = if b.opponent == "human" {
        Seat::Open
    } else {
        let o = app.opponents.get(&b.opponent).ok_or_else(|| bad(format!("unknown opponent {:?}", b.opponent)))?;
        Seat::Bot { spec: o.spec.clone(), name: o.name.clone() }
    };
    let mine = Seat::Player { id: p.id, name: p.name };
    let first = match b.seat.as_deref().unwrap_or("random") {
        "first" => true,
        "second" => false,
        "random" => rooms::random_id(1).as_bytes()[0] % 2 == 0,
        s => return Err(bad(format!("unknown seat choice {s:?}"))),
    };
    let seats = if first { [mine, other] } else { [other, mine] };
    let room = app.create_game(sets, seats).map_err(internal)?;
    room.kick_bot(&app);
    Ok(Json(json!({ "id": room.id, "seat": if first { 0 } else { 1 } })))
}

async fn list_games(State(app): State<Arc<App>>, headers: HeaderMap) -> ApiResult<Json<serde_json::Value>> {
    let p = require_player(&app, &headers)?;
    Ok(Json(json!(app.db.games_of(&p.id, 200).map_err(internal)?)))
}

async fn join_game(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let p = require_player(&app, &headers)?;
    let room = find_room(&app, &id)?;
    let seat = room.join(&app, &p).map_err(bad)?;
    Ok(Json(json!({ "seat": seat })))
}

async fn replay(State(app): State<Arc<App>>, Path(id): Path<String>) -> ApiResult<Json<serde_json::Value>> {
    let room = find_room(&app, &id)?;
    let (replay, seats, resigned) =
        room.finished_replay().ok_or(ApiError(StatusCode::FORBIDDEN, "replays open when the game ends".into()))?;
    Ok(Json(json!({ "id": id, "replay": replay, "seats": seats, "resigned": resigned })))
}

#[derive(Deserialize)]
struct WsQuery {
    token: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMsg {
    Act { action: u16, n: Option<usize> },
    PlayAll { n: Option<usize> },
    Resign,
}

async fn ws(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    Query(q): Query<WsQuery>,
    upgrade: WebSocketUpgrade,
) -> ApiResult<Response> {
    let room = find_room(&app, &id)?;
    let me = player(&app, q.token.as_deref())?;
    Ok(upgrade.on_upgrade(move |socket| client(app, room, me, socket)))
}

async fn client(app: Arc<App>, room: Arc<Room>, me: Option<Player>, socket: WebSocket) {
    let (mut tx, mut rx) = socket.split();
    let mut version = room.subscribe();
    let mut seat = room.with(|st| st.seat_of(me.as_ref()));
    room.set_online(seat, 1);
    room.kick_bot(&app);
    let mut ping = tokio::time::interval(Duration::from_secs(25));
    let mut dirty = true;
    loop {
        if dirty {
            dirty = false;
            let now_seat = room.with(|st| st.seat_of(me.as_ref()));
            if now_seat != seat {
                room.set_online(seat, -1);
                room.set_online(now_seat, 1);
                seat = now_seat;
            }
            let msg = room.with(|st| st.message(&room.id, seat));
            if tx.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
        tokio::select! {
            changed = version.changed() => {
                if changed.is_err() { break; }
                version.borrow_and_update();
                dirty = true;
            }
            _ = ping.tick() => {
                if tx.send(Message::Ping(Default::default())).await.is_err() { break; }
            }
            incoming = rx.next() => {
                let text = match incoming {
                    Some(Ok(Message::Text(t))) => t,
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                    Some(Ok(_)) => continue,
                };
                let result = match (&me, serde_json::from_str::<ClientMsg>(&text)) {
                    (_, Err(e)) => Err(format!("bad message: {e}")),
                    (None, _) => Err("spectators cannot play".to_string()),
                    (Some(p), Ok(m)) => {
                        let (n, mv) = match m {
                            ClientMsg::Act { action, n } => (n, Move::Action(action)),
                            ClientMsg::PlayAll { n } => (n, Move::PlayAll),
                            ClientMsg::Resign => (None, Move::Resign),
                        };
                        room.play(&app, p, n, mv)
                    }
                };
                if let Err(message) = result {
                    let msg = serde_json::to_string(&ServerMsg::Error { message }).unwrap();
                    if tx.send(Message::Text(msg.into())).await.is_err() { break; }
                    dirty = true;
                }
            }
        }
    }
    room.set_online(seat, -1);
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let args = Args::parse();
    let db = Db::open(&args.db).unwrap_or_else(|e| panic!("opening {}: {e}", args.db.display()));
    let opponents = App::load_opponents(&args.models);
    let app = Arc::new(App::new(db, opponents, Duration::from_millis(args.bot_delay_ms)));

    if !args.assets.join("cards").is_dir() {
        tracing::warn!("{} has no card art; run tools/fetch_assets.py (the client falls back to text cards)", args.assets.display());
    }
    if !args.web.join("index.html").is_file() {
        tracing::warn!("{} has no built client; run `npm run build` in web/", args.web.display());
    }

    let api = Router::new()
        .route("/players", post(create_player))
        .route("/me", get(me).put(rename))
        .route("/catalog", get(catalog))
        .route("/opponents", get(list_opponents))
        .route("/games", get(list_games).post(create_game))
        .route("/games/{id}/join", post(join_game))
        .route("/games/{id}/replay", get(replay))
        .route("/games/{id}/ws", get(ws))
        .layer(SetResponseHeaderLayer::overriding(header::CACHE_CONTROL, HeaderValue::from_static("no-store")));

    let assets = Router::new().fallback_service(ServeDir::new(&args.assets)).layer(
        SetResponseHeaderLayer::overriding(header::CACHE_CONTROL, HeaderValue::from_static("public, max-age=604800")),
    );

    let spa = ServeDir::new(&args.web).fallback(ServeFile::new(args.web.join("index.html")));
    let router = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .nest("/api", api)
        .nest_service("/assets", assets)
        .fallback_service(spa)
        .with_state(app);

    let listener = tokio::net::TcpListener::bind(&args.bind).await.unwrap_or_else(|e| panic!("binding {}: {e}", args.bind));
    tracing::info!("listening on http://{}", args.bind);
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .unwrap();
}
