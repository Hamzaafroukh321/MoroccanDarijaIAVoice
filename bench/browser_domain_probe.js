// Server-rendered temporary third-domain fixture; no microphone/provider calls.
async (page) => {
  const url='http://127.0.0.1:8000/?domain=service-desk&mode=demo';
  await page.route(url,route=>route.fulfill({path:'output/playwright/custom_domain_diagnostic.html',contentType:'text/html'}));
  await page.goto(url);
  const selected=await page.evaluate(()=>({
    domain:document.getElementById('domain').value,
    mode:document.getElementById('mode').value,
    options:[...document.getElementById('domain').options].map(option=>({id:option.value,title:option.textContent})),
    heading:document.getElementById('session-title').textContent,
  }));
  if(selected.domain!=='service-desk'||selected.mode!=='demo'||selected.options.length!==3) throw new Error(JSON.stringify(selected));
  if(selected.options[2].title!=='Counter service diagnostic') throw new Error('Custom config title missing.');
  const rendered=await page.evaluate(()=>{
    showSlots({items:['counter_a','counter_b'],date:'2026-09-15',time:'10:30'});
    return document.getElementById('slots').textContent;
  });
  if(!rendered.includes('Counter A, Counter B')||rendered.includes('Pizza')) throw new Error('Flat items field used collection rendering.');
  await page.screenshot({path:'output/playwright/custom_domain_diagnostic.png',fullPage:true});
  await page.getByLabel('Domain',{exact:true}).selectOption('clinic');
  await page.waitForURL('**/?domain=clinic&mode=demo');
  const navigation=await page.evaluate(()=>({domain:document.getElementById('domain').value,
    mode:document.getElementById('mode').value,url:location.href}));
  if(navigation.domain!=='clinic'||navigation.mode!=='demo') throw new Error('Domain switch lost demo mode.');
  const report={purpose:'Real browser selector and controlled flat-field rendering with server-rendered temporary third-domain HTML; normal navigation to shipped clinic. No speech/device quality claim.',provider_requests:0,selected,rendered,navigation};
  await page.evaluate(value=>{window.__domainProbeReport=value;},report);
  return report;
}
