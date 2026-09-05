async (page) => {
await page.goto('http://127.0.0.1:8000/?domain=clinic&mode=demo');
await page.waitForSelector('#demo-note:not([hidden])');
const report = await page.evaluate(() => ({
  location: location.href,
  domain: document.querySelector('#domain').value,
  mode: document.querySelector('#mode').value,
  label: document.querySelector('#demo-voice-label').textContent,
  button: document.querySelector('#record').textContent,
  setup: document.querySelector('#setup').textContent,
}));
if (!report.label.includes('MoulSot on this computer') || report.mode !== 'demo') {
  throw new Error('Local MoulSot configuration is not visible in the demo');
}
await page.evaluate(value => { window.__LOCAL_ASR_REPORT = value; }, report);
await page.screenshot({path: 'output/playwright/local_moulsot_clinic.png', fullPage: true});
return report;
}
