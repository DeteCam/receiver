// 화면을 머리 없는 크로미움으로 열어 오류를 모으고 그림을 찍는다.  node shot.js <html> <out.png> [폭] [dark|light] [스크립트]
const { chromium } = require('playwright');
(async () => {
  const [file, out, width, theme, script] = process.argv.slice(2);
  const browser = await chromium.launch(process.env.SWIFT ? { args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] } : { headless: false, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: +(width || 1180), height: 900 }, deviceScaleFactor: 1, colorScheme: theme === 'dark' ? 'dark' : 'light' });
  const errs = [];
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errs.push(m.type() + ': ' + m.text()); });
  page.on('pageerror', (e) => errs.push('pageerror: ' + e.message));
  await page.goto('file://' + require('path').resolve(file));
  await page.waitForTimeout(1500);
  if (script) { const r = await page.evaluate(script); if (r !== undefined) console.log('결과:', JSON.stringify(r)); await page.waitForTimeout(800); }
  const h = await page.evaluate(() => document.documentElement.scrollHeight);
  const ov = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  await page.screenshot({ path: out, fullPage: true });
  console.log('높이', h, '가로 넘침', ov, '오류', errs.length); errs.slice(0, 12).forEach((e) => console.log('  ' + e));
  await browser.close();
})();
