var http = require('http');
var url = require('url');
var fs = require('fs');
var txtwiki = require('./txtwiki.js');

var a_items = [];
var new_items = [];
var len = 0;

fs.readFile( __dirname + '/data.json', function (err, data) {
  if (err) {
    throw err;
  }
  // console.log(data.toString());
  a_items = JSON.parse(data.toString());
  len = a_items.length;
  console.log(a_items.length);
  process();

});

function process(){
  // interval = setInterval(f, 10);
  for (var i = 0; i < len; i++) {
    f();
  }
  require('fs').writeFile(

    './my.json',

    JSON.stringify(new_items),

    function (err) {
        if (err) {
            console.error('Crap happens');
        }
    }
);
  // console.log(new_items.length);
}

var ptexts = [];

function f() {
    item = a_items.pop();
    ptext = txtwiki.parseWikitext(item.txt);
    // ptexts.push(ptext);
    // ptext = "";
    new_items.push({
        page_id: item.page_id,
        parsed_text: ptext,
        category: item.cat
    });
    // console.log('.');
    // if(new_items.length == len){
    //     clearInterval(interval);
    // }
    console.log(a_items.length);

}

// var a_items = JSON.parse(reader.result);


// //create a server object:
// http.createServer(function (req, res) {
//   res.write('Hello World!'); //write a response to the client
//   res.end(); //end the response
// }).listen(8080); //the server object listens on port 8080

