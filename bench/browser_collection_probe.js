async (page) => {
  // Engineering fixture only: real DOM rendering, mocked configuration, microphone and transport.
  // Run in a fresh named Playwright CLI context; this installs a page initialization script.
  const configuration = {
    domain_id: 'collection_fixture', domain_title: 'Equipment reservation fixture',
    demo_supported: true, demo_state_kind: 'configured_collection_scoped', demo_issues: [],
    demo_collection: {name:'reservations',label:'Reservation',root_slots:['return_date'],item_slots:['asset','quantity'],min_items:1,max_items:10},
    demo_labels: {reservations:'Reservations',asset:'Equipment',quantity:'Quantity',return_date:'Return date'},
    demo_values: {camera:'Camera',tripod:'Tripod',light:'Light'},
    demo_ui: {title:'Reserve equipment · browser fixture',note:'Mocked interface check. No reservation is made.',state_title:'Saved reservations',state_empty:'Your equipment will appear here.',completion_text:'Fixture completed. No reservation was made.'},
    domains: [{id:'collection_fixture',title:'Equipment reservation fixture'}],
  };
  await page.route('http://127.0.0.1:8000/?domain=clinic&mode=demo', async route => {
    const response = await route.fetch();
    let html = await response.text();
    html = html.replace(/(<script id="capture-config"[^>]*>)([\s\S]*?)(<\/script>)/, (_, before, value, after) =>
      before + JSON.stringify({...JSON.parse(value),...configuration}) + after);
    await route.fulfill({response,body:html});
  });
  await page.addInitScript(() => {
    const fixture = window.__collectionFixture = {sockets:[],tracks:[]};
    window.AudioContext = class {
      constructor() {this.sampleRate=16000;this.state='running';this.destination={};this.audioWorklet={addModule:async()=>{}};}
      async resume() {} async close() {this.state='closed';}
      createMediaStreamSource() {return {connect(){},disconnect(){}};}
    };
    window.AudioWorkletNode = class {
      constructor() {this.port={close(){},onmessage:null,postMessage:value=>{
        if(value.type==='stop') setTimeout(()=>this.port.onmessage?.({data:{type:'finished',samples:0}}),0);
      }};}
      connect() {} disconnect() {}
    };
    Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:async()=>{
      const track={stopped:false,onended:null,stop(){this.stopped=true;}};
      fixture.tracks.push(track);
      return {getTracks:()=>[track],getAudioTracks:()=>[track]};
    }});
    window.WebSocket = class {
      static OPEN=1;
      constructor(){this.readyState=1;this.bufferedAmount=0;fixture.sockets.push(this);setTimeout(()=>this.emit({type:'ready'}),0);}
      emit(value){this.onmessage?.({data:JSON.stringify(value)});}
      send(data){const value=JSON.parse(data);
        if(value.type==='start') setTimeout(()=>{
          this.emit({type:'started',session_id:'collection-fixture'});
          this.emit({type:'state',state:'LISTENING',slots:{reservations:[],return_date:'2026-09-20'}});
        },0);
        if(value.type==='stop') setTimeout(()=>this.emit({type:'done',status:'interrupted',slots:{}}),0);
      }
      close(){this.readyState=3;}
    };
  });
  await page.goto('http://127.0.0.1:8000/?domain=clinic&mode=demo');
  await page.locator('#record').click();
  await page.waitForFunction(()=>window.__collectionFixture.sockets.length===1 && document.querySelector('#status').textContent==='Listening');
  const emptyText=await page.locator('#slots').innerText();
  if(!emptyText.includes('No items added yet.') || !emptyText.toLowerCase().includes('return date')) throw new Error('Empty collection or independent root field missing.');
  await page.evaluate(()=>window.__collectionFixture.sockets[0].emit({type:'state',state:'LISTENING',
    slots:{reservations:[{id:2,asset:'camera',quantity:1},{id:3,asset:'tripod',quantity:2}],return_date:'2026-09-20'},
    pending_clarification:{id:'question-1',kind:'item_reference',slot:'quantity',item_ids:[2,3]},
    pending_proposal:{state:{reservations:[{id:2,asset:'camera',quantity:1},{id:3,asset:'tripod',quantity:2},{id:4,asset:'light',quantity:1}],return_date:'2026-09-20'}}}));
  await page.locator('#proposal-panel:not([hidden])').waitFor();
  const saved=await page.locator('#slots').innerText();
  const proposed=await page.locator('#proposed-slots').innerText();
  if(!saved.includes('Camera')||!saved.includes('Tripod')||saved.includes('Light')||!proposed.includes('Light')) throw new Error('Proposed addition leaked into saved details or was omitted.');
  const labels=await page.locator('#slots dt[data-item-id]').evaluateAll(nodes=>nodes.map(node=>({id:node.dataset.itemId,label:node.textContent})));
  if(JSON.stringify(labels)!==JSON.stringify([{id:'2',label:'Reservation 1'},{id:'3',label:'Reservation 2'}])) throw new Error('Stable IDs or visible ordinals were changed.');
  if(!proposed.toLowerCase().includes('return date') || !proposed.includes('2026-09-20')) throw new Error('Root field omitted from preview.');
  if(!(await page.locator('#proposal-note').innerText()).includes('not confirmed')) throw new Error('Missing pending disclosure.');
  for(const [name,width,height] of [['desktop',1365,1100],['mobile',390,844]]) {
    await page.setViewportSize({width,height});
    if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)) throw new Error(`${name} overflow.`);
    await page.screenshot({path:`output/playwright/collection_${name}.png`,fullPage:true});
  }
  await page.evaluate(()=>window.__collectionFixture.sockets[0].emit({type:'state',state:'CONFIRMING',
    slots:{reservations:[{id:2,asset:'camera',quantity:1},{id:3,asset:'tripod',quantity:2},{id:4,asset:'light',quantity:1}],return_date:'2026-09-20'},pending_proposal:null}));
  if(await page.locator('#proposal-panel').isVisible() || !(await page.locator('#slots').innerText()).includes('Light')) throw new Error('Commit did not clear preview or update saved values.');
  await page.locator('#record').click();
  await page.waitForFunction(()=>window.__collectionFixture.tracks.every(track=>track.stopped));
  return {kind:'real_browser_mocked_configuration_microphone_and_socket',provider_calls:0,checks:['empty collection help with independent root field','configured field/value labels','stable IDs after deletion with contiguous display ordinals','pending addition separate from saved state','root field preserved in preview','pending disclosure','desktop/mobile no overflow','commit clears preview','microphone released']};
}
