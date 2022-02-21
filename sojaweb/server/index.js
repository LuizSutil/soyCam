const PORT = 8000;
const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");

var fs = require('fs');

const app = express();
app.use(express.json());

var count = 0

app.get("/stream", (req, res) => {

  res.set({
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",

    // enabling CORS
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers":
      "Origin, X-Requested-With, Content-Type, Accept",
  })

  setInterval(() => {
    try {  
      var data = fs.readFileSync('../../count.txt', 'utf8');
        count = data.toString();    
      } catch(e) {
          console.log('Error:', e.stack);
    }
    res.write(`data: ${count}\n\n`)
  }, 50)
})

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));



app.listen(PORT, () => console.log(`The server is listening on port ${PORT}`));

 


