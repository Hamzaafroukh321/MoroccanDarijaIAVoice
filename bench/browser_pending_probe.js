// Controlled state-event rendering; no microphone or provider requests.
async (page) => {
  const results=[];
  for (const domain of ['clinic', 'pizza']) {
    for (const width of [1100, 390]) {
      await page.setViewportSize({width, height: 900});
      await page.goto(`http://127.0.0.1:8000/?domain=${domain}&mode=demo`);
      const view=await page.evaluate(domain => {
        const committed=domain==='clinic' ? {} : {items:[{id:1,quantity:1,size:'large',toppings:[]}]};
        const proposed=domain==='clinic' ? {time:'10:30'} : {items:[{id:1,quantity:1,size:'small',toppings:[]}]};
        current={phase:'recording',mode:'demo'};
        const event={type:'state',state:'SPEAKING',slots:committed,
          pending_proposal:{state:proposed},pending_clarification:null};
        handleVoiceEvent(current,event);
        window.__pendingFixture={committed,proposed};
        addMessage('Assistant',domain==='clinic' ? 'هاد الاختيار ما كاينش فالتجربة. الاختيارات: الطبيب ألف، الطبيب باء.' : 'شنو بغيتي تشرب؟');
        return {visible:!document.getElementById('proposal-panel').hidden,
          currentText:document.getElementById('slots').textContent,
          proposedText:document.getElementById('proposed-slots').textContent,
          note:document.getElementById('proposal-note').textContent,
          noHorizontalOverflow:document.documentElement.scrollWidth<=window.innerWidth};
      },domain);
      if (!view.visible || !view.noHorizontalOverflow || !view.note.includes('Waiting for your answer')) throw new Error(JSON.stringify(view));
      if (domain==='clinic' && (view.currentText.includes('10:30') || !view.proposedText.includes('10:30'))) throw new Error('Held clinic time was lost or presented as current.');
      if (domain==='pizza' && (!view.currentText.includes('large') || !view.proposedText.includes('small'))) throw new Error('Pizza current/proposed size was mixed.');
      await page.screenshot({path:`output/playwright/proposed_${domain}_${width}.png`,fullPage:true});
      const transitions=await page.evaluate(() => {
        const {committed,proposed}=__pendingFixture;
        handleVoiceEvent(current,{type:'state',state:'LISTENING',slots:committed,pending_proposal:null});
        const discarded=document.getElementById('proposal-panel').hidden && document.getElementById('proposed-slots').children.length===0;
        handleVoiceEvent(current,{type:'state',state:'SPEAKING',slots:committed,pending_proposal:{state:proposed}});
        handleVoiceEvent(current,{type:'state',state:'CONFIRMING',slots:proposed,pending_proposal:null});
        const resolved=document.getElementById('proposal-panel').hidden && document.getElementById('slots').children.length>0;
        handleVoiceEvent(current,{type:'state',state:'SPEAKING',slots:committed,pending_proposal:{state:proposed}});
        handleVoiceEvent(current,{type:'state',state:'DONE',slots:committed,pending_proposal:{state:proposed}});
        const ended=document.getElementById('proposal-panel').hidden;
        current=null; idle();
        return {discarded,resolved,ended};
      });
      if (!Object.values(transitions).every(Boolean)) throw new Error(JSON.stringify(transitions));
      results.push({domain,width,...view,...transitions});
    }
  }
  const report={kind:'browser_pending_details_diagnostic',provider_requests:0,
    input:'controlled state events against actual frontend; draft assistant text',results};
  await page.evaluate(value=>{window.__pendingProbeReport=value;},report);
  return report;
}
