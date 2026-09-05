// Uses HTML produced by server.index with an invalid provider in a child process.
// The real .env and running server configuration are never modified.
async (page) => {
  const url='http://127.0.0.1:8000/?domain=clinic&mode=capture';
  await page.route(url,route=>route.fulfill({path:'output/playwright/setup_invalid_clinic.html',contentType:'text/html'}));
  await page.goto(url);
  const capture=await page.evaluate(()=>({status:document.getElementById('status').textContent,
    enabled:!document.getElementById('record').disabled,mode:document.getElementById('mode').value}));
  if(capture.status!=='Ready to record'||!capture.enabled||capture.mode!=='capture') throw new Error(JSON.stringify(capture));
  await page.getByLabel('Session',{exact:true}).selectOption('demo');
  const demo=await page.evaluate(()=>({status:document.getElementById('status').textContent,
    disabled:document.getElementById('record').disabled,message:document.getElementById('message').textContent}));
  if(demo.status!=='Voice setup needed'||!demo.disabled||!demo.message.includes('DEMO_TTS_PROVIDER')) throw new Error(JSON.stringify(demo));
  await page.screenshot({path:'output/playwright/invalid_provider_setup.png',fullPage:true});
  await page.getByLabel('Session',{exact:true}).selectOption('capture');
  const restored=await page.evaluate(()=>document.getElementById('status').textContent==='Ready to record'&&!document.getElementById('record').disabled);
  if(!restored)throw new Error('Capture did not recover after viewing demo setup issue.');
  const report={kind:'browser_invalid_demo_provider_diagnostic',provider_requests:0,
    source:'child-process server-rendered HTML with invalid provider; mocked page response, real frontend assets',capture,demo,restored};
  await page.evaluate(value=>{window.__setupProbeReport=value;},report);
  return report;
}
