import urllib.parse
import zlib
import base64

test_xml = '<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/><mxCell id="2" value="Hello World" vertex="1" parent="1"><mxGeometry x="10" y="10" width="100" height="50" as="geometry"/></mxCell></root></mxGraphModel>'

# Standard Draw.io encoder
quoted = urllib.parse.quote(test_xml, safe='')
compressor = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
compressed = compressor.compress(quoted.encode('latin1')) + compressor.flush()
b64 = base64.b64encode(compressed).decode('ascii')

# Standard Draw.io decoder
raw = base64.b64decode(b64)
decomp = zlib.decompress(raw, -15)
result = urllib.parse.unquote(decomp.decode('latin1'))
print('Roundtrip match:', result == test_xml)
