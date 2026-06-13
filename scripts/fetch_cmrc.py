import urllib.request, json, ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
req = urllib.request.Request('https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_dev.json')
with urllib.request.urlopen(req, context=ctx) as f:
    data = json.load(f)
    print(json.dumps(data['data'][0]['paragraphs'][0]['qas'][0], ensure_ascii=False))
    print(data['data'][0]['paragraphs'][0]['context'])
