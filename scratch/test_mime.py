import mimetypes
mimetypes.init()
print(f"JS MIME: {mimetypes.guess_type('test.js')[0]}")
mimetypes.add_type('application/javascript', '.js', True)
print(f"JS MIME AFTER: {mimetypes.guess_type('test.js')[0]}")
