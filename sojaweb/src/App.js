import logo from './logo.svg';
import { useEffect,useState } from 'react';
import './App.css';

function App() {


  const [soyCount,setSoyCount] = useState(0)
  useEffect(() => {
    let eventSource = new EventSource("http://localhost:8000/stream")
    eventSource.onmessage = e => setSoyCount(JSON.parse(e.data))
  },  [])

  return (
    <div className="App">
      <header className="App-header">
        <h1>{soyCount}</h1>
      </header>
    </div>
  );
}

export default App;
