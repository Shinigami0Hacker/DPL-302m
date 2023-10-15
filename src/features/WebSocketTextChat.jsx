export default function WebSocketTextChat(){
    
    let url = `ws:://${window.location.host}/w`;

    const chatSocket = new WebSocket(url);

    chatSocket.onmessage = function(e){
        let data = JSON.parse(e.data)
        console.log('Data', data)
    }
}
