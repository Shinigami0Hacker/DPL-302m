import { useEffect, useRef, useState } from "react"
import { faArrowAltCircleRight, faCamera, faXmark } from "@fortawesome/free-solid-svg-icons"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import Webcam from "react-webcam"

const ChatPageContent = () => {
    useEffect((e) => {

    }, [])

    const [message, setMessage] = useState('')
    const [textBox, setTextBox] = useState([])
    const [openVideo, setOpenVideo] = useState(false)

    const webRef = useRef(null)

    const handleRecieve = async (e) => {

    }
    
    let user = {
        role: "Doctor"
    }

    const handleCreateRoom = (e) => {
        e.preventDefault()

    }

    const handleSend = (e) => {
        e.preventDefault()

        const date_obj = new Date;

        const time_string = `${date_obj.getHours}:${date_obj.getMinutes}:${date_obj.getSeconds()}`
        if (message !== ''){
            setTextBox([...textBox, {message: message, role: user.role, time_stamp: time_string}])
        }
        setMessage('')
    }

    const closeVideoModal = () => {
        setOpenVideo(false)
    }
    
    return ( 
        <div className="bg-indigo-600 flex h-screen items-center justify-center relative">
            {openVideo ? 
            <section className="p-8 bg-slate-400 rounded-lg">
                <FontAwesomeIcon onClick= {closeVideoModal} icon={faXmark}/>
                <Webcam ref = {webRef}/>
            </section>:
            <></>}
            <section className="transform ease-linear relative w-1/2 h-3/4 shadow-lg bg-white rounded-lg min-h-min">
                    <section className="m-auto mt-6 overflow-y-auto h-4/5 w-11/12 border border-indigo-600 rounded-lg">
                        {textBox.map((box, index) => {
                            return (
                                <div key = {index} className=" flex justify-end">
                                    <div className="bg-slate-400 p-2 mt-4 mr-4 rounded-md">
                                        {box}
                                    </div>
                                </div>
                            )
                        })}
                    </section>
                    <section className="flex mt-5 items-center justify-center">
                        <input value = {message} onChange={(event) => setMessage(event.target.value)} className="h-8 w-5/6 p-2 mr-4 border border-indigo-600 rounded-md hover:none focus:border-red-600" type="text"/>
                        <button onClick={handleSend} className="h-8 cursor-pointer"><FontAwesomeIcon icon={faArrowAltCircleRight} size="2x" color="#3949AB" /></button>
                    </section> 
                    <section className="mt-1 p-1  flex justify-center items-center">
                        <button class="bg-red-500 hover:bg-red-400 text-white font-bold py-2 px-4 border-b-4 border-red-700 hover:border-red-500 rounded">Process</button>
                        <button className="mr-4 ml-4"><FontAwesomeIcon size="2x"onClick = {(event) => {setOpenVideo(!openVideo)}} className="cursor-pointer" icon={faCamera}/></button>
                        <button class="bg-blue-500 hover:bg-blue-400 text-white font-bold py-2 px-4 border-b-4 border-blue-700 hover:border-blue-500 rounded">Import</button>
                    </section>
            </section>
        </div>
    )
}
export default ChatPageContent