import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob, PresentationFile} from 'file:///C:/Users/sooyah/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const base = path.join(root, '..', 'wt-ppt-latest/ppt/DCC2026_예선_결과정리.pptx');
const out = path.join(root, 'outputs/submission_deck');
await fs.mkdir(out, {recursive:true});
const p = await PresentationFile.importPptx(await FileBlob.load(base));
const content = await p.inspect({select:'$.slides[*]',include:['id','index','elements.id','elements.kind','elements.text','notes.text']});
await fs.writeFile(path.join(out, 'source-inspect.ndjson'), content.ndjson);
for (const n of [9,10,14,15,16,17]) {
  const slide = p.slides.getItem(n-1);
  const png = await p.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(out, `before-${n}.png`), new Uint8Array(await png.arrayBuffer()));
}
console.log(`Imported ${p.slides.items.length} slides; source inspection/render saved.`);
