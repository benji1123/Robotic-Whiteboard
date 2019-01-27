var form = new FormData();
form.append("file", "D:\\School\\Deltahacks V\\DeltaDraw\\Delta_Draw\\image_to_draw_preview.png");

var settings = {
  "async": true,
  "crossDomain": true,
  "url": "https://e1eea922.ngrok.io",
  "method": "POST",
  "headers": {
    "cache-control": "no-cache",
    "Postman-Token": "4447e470-e223-453a-90ce-cbd2c2d2269a"
  },
  "processData": false,
  "contentType": false,
  "mimeType": "multipart/form-data",
  "data": form
}

$.ajax(settings).done(function (response) {
  console.log(response);
});