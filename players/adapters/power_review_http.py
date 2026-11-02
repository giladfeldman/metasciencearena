"""One HTTP request in a killable process; credentials travel over stdin only."""
import json
import os
import subprocess
import sys
from urllib.parse import urlsplit

_REQUEST = r'''
import json,sys,requests
data=json.load(sys.stdin)
try:
 with requests.post(data['url'],json=data['body'],headers={'Authorization':'Bearer '+data['key'],'Content-Type':'application/json'},timeout=data['timeout'],stream=True,allow_redirects=False) as response:
  raw=bytearray()
  if 200<=response.status_code<300:
   for chunk in response.iter_content(65536):
    raw.extend(chunk)
    if len(raw)>1024*1024:break
  if len(raw)>1024*1024:print(json.dumps({'error':'RESPONSE_TOO_LARGE'}))
  else:print(json.dumps({'status':response.status_code,'body':raw.decode('utf-8')}))
except Exception:print(json.dumps({'error':'TRANSPORT_FAILED'}))
'''

def bounded_post(url,key,body,timeout_s,*,loopback_test=False):
    parts=urlsplit(url)
    permitted=parts.scheme=='https' and parts.hostname in {'api.groq.com','integrate.api.nvidia.com'}
    if loopback_test:permitted=permitted or (parts.scheme=='http' and parts.hostname=='127.0.0.1')
    if not permitted or parts.username or parts.password:raise ValueError('Unapproved evaluation endpoint')
    timeout=min(float(timeout_s),55)
    if timeout<=0:raise ValueError('Positive transport deadline required')
    environment={key:value for key,value in os.environ.items() if key.upper() in {'SYSTEMROOT','WINDIR','PATH','TMP','TEMP','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE'}}
    try:
        result=subprocess.run([sys.executable,'-I','-c',_REQUEST],input=json.dumps({'url':url,'key':key,'body':body,'timeout':timeout}),capture_output=True,text=True,timeout=timeout,env=environment)
    except subprocess.TimeoutExpired:
        # run() kills and waits for its child, closing the request before returning.
        raise TimeoutError('Provider transport deadline exceeded; response withheld') from None
    try:data=json.loads(result.stdout)
    except (ValueError,TypeError):raise RuntimeError('Provider transport failed; diagnostics withheld') from None
    if result.returncode or data.get('error'):raise RuntimeError('Provider transport failed: '+str(data.get('error','CHILD_FAILED')))
    return data
