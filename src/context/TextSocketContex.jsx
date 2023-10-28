import { createContext, useContext, useEffect, useState } from "react";

const TextSocketContext = createContext()

export default TextSocketContext

export const TextSocketProvider = ( { children } ) => {
    
    
    
    const [textBox, setTextBox] = useState([])
    
    let url = `ws://127.0.0.1:8000/ws/socket-server/`;
    
    const textChatSocket = new WebSocket(url);
    useEffect(() => {
        const websocketConnectionTimeout = setTimeout(() => {
            console.log(`%c Connection timeout. Failed to connect to ${url}`, 'color: red');
        }, 5000)

        textChatSocket.addEventListener("open", () => {
            clearTimeout(websocketConnectionTimeout);
            console.log(`%c Etablish connection with ${url}`, 'color: green')
        })
        
        textChatSocket.onmessage = function (e){
            let data = JSON.parse(e.data)
            let date_object = new Date
            let time_string = `${date_object.getHours}-${date_object.getMinutes}-${date_object.getSeconds}`

            setTextBox([...textBox, {message: data.message, role: 'Patient', time_stamp: time_string}])
        }
        textChatSocket.addEventListener("close", () => {
            console.log(`Clear cached, close socket at ${url}`)
        })
    })

    const context = {
        socket: textChatSocket
    }
    
    return (
        <TextSocketContext.Provider>
            { children }
        </TextSocketContext.Provider>
    )
}