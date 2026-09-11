// Serve the real committed dashboard bundle and CSS. Tests own local API replies.
import http from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(fileURLToPath(new URL("../../", import.meta.url)));
const types = { ".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".svg": "image/svg+xml" };
http.createServer(async (request, response) => {
    const url = new URL(request.url, "http://localhost");
    const relative = url.pathname === "/" ? "templates/index.html" : url.pathname.slice(1);
    const file = path.resolve(root, relative);
    if (!file.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
    try {
        response.setHeader("Content-Type", types[path.extname(file)] || "application/octet-stream");
        response.end(await readFile(file));
    } catch { response.writeHead(404).end(); }
}).listen(4173, "127.0.0.1");
