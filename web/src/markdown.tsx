import type { ReactNode } from "react";

const BASE = "/ai/";

function safeUrl(url: string): string | null {
  const t = url.trim();
  if (/^(javascript|data):/i.test(t)) return null;
  if (/^(https?:|\/)/i.test(t)) return t;
  return BASE + t.replace(/^\.\//, "");
}

/** Inline markdown: images, links, bold, italics, code. */
function inline(src: string, key: string): ReactNode[] {
  const re = /!\[([^\]]*)\]\(([^)]+)\)|\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`/g;
  const out: ReactNode[] = [];
  let last = 0;
  let i = 0;
  for (const m of src.matchAll(re)) {
    const at = m.index ?? 0;
    if (at > last) out.push(src.slice(last, at));
    const k = `${key}-${i++}`;
    if (m[1] != null) {
      const href = safeUrl(m[2]);
      out.push(href ? <img key={k} src={href} alt={m[1]} /> : m[0]);
    } else if (m[3] != null) {
      const href = safeUrl(m[4]);
      out.push(
        href ? (
          <a key={k} href={href} target={href.startsWith("http") ? "_blank" : undefined} rel="noreferrer">
            {m[3]}
          </a>
        ) : (
          m[3]
        ),
      );
    } else if (m[5] != null) {
      out.push(<strong key={k}>{m[5]}</strong>);
    } else if (m[6] != null) {
      out.push(<em key={k}>{m[6]}</em>);
    } else {
      out.push(<code key={k}>{m[7]}</code>);
    }
    last = at + m[0].length;
  }
  if (last < src.length) out.push(src.slice(last));
  return out;
}

/** A small subset of markdown: headings, lists, quotes, code fences, links, and images. */
export function renderMarkdown(source: string): ReactNode[] {
  const lines = source.replace(/<!--[\s\S]*?-->/g, "").replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    if (line.startsWith("```")) {
      const buf: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) buf.push(lines[i++]);
      if (i < lines.length) i++;
      blocks.push(<pre key={blocks.length}><code>{buf.join("\n")}</code></pre>);
      continue;
    }
    const heading = /^(#{1,3})\s+(.*)$/.exec(line);
    if (heading) {
      const text = inline(heading[2], `h${blocks.length}`);
      const Tag = `h${heading[1].length}` as "h1" | "h2" | "h3";
      blocks.push(<Tag key={blocks.length}>{text}</Tag>);
      i++;
      continue;
    }
    if (/^(-{3,}|\*{3,})$/.test(line.trim())) {
      blocks.push(<hr key={blocks.length} />);
      i++;
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i])) {
        items.push(<li key={items.length}>{inline(lines[i].replace(/^[-*]\s+/, ""), `ul${blocks.length}-${items.length}`)}</li>);
        i++;
      }
      blocks.push(<ul key={blocks.length}>{items}</ul>);
      continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        items.push(<li key={items.length}>{inline(lines[i].replace(/^\d+\.\s+/, ""), `ol${blocks.length}-${items.length}`)}</li>);
        i++;
      }
      blocks.push(<ol key={blocks.length}>{items}</ol>);
      continue;
    }
    if (line.startsWith(">")) {
      const buf: string[] = [];
      while (i < lines.length && lines[i].startsWith(">")) buf.push(lines[i++].replace(/^>\s?/, ""));
      blocks.push(<blockquote key={blocks.length}>{inline(buf.join(" "), `q${blocks.length}`)}</blockquote>);
      continue;
    }
    const buf: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].startsWith("```") &&
      !lines[i].startsWith(">") &&
      !/^#{1,3}\s/.test(lines[i]) &&
      !/^[-*]\s+/.test(lines[i]) &&
      !/^\d+\.\s+/.test(lines[i]) &&
      !/^(-{3,}|\*{3,})$/.test(lines[i].trim())
    ) {
      buf.push(lines[i++]);
    }
    blocks.push(<p key={blocks.length}>{inline(buf.join(" "), `p${blocks.length}`)}</p>);
  }
  return blocks;
}
