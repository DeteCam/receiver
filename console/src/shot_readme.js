// README 의 시뮬레이터 그림(docs/images/34_시뮬레이터.png): 첫 화면에서 결과 카드 아래까지 2배로 찍는다.  node shot_readme.js <만든 html> <저장할 png>   (playwright 가 있어야 한다)
const { chromium } = require('playwright');
(async () => {
  const [file, out] = process.argv.slice(2);
  const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 1280, height: 1000 }, deviceScaleFactor: 2, colorScheme: 'light' });
  const errs = [];
  p.on('pageerror', (e) => errs.push('pageerror: ' + e.message));
  await p.goto('file://' + require('path').resolve(file));
  await p.waitForTimeout(4000);
  const r = await p.evaluate(() => { const bs = ['result', 'verdict'].map((id) => document.getElementById(id).getBoundingClientRect()); return { top: bs[0].top, bottom: Math.max(...bs.map((b) => b.bottom)), h: document.documentElement.scrollHeight }; });
  const H = Math.min(Math.ceil(r.bottom + 18), r.h);
  await p.screenshot({ path: out, clip: { x: 0, y: 0, width: 1280, height: H } });
  console.log(JSON.stringify(r), 'H', H, '오류', errs.length, errs.slice(0, 3));
  await b.close();
})();
