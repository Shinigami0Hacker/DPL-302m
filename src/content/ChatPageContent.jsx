import { useEffect, useRef, useState } from "react"
import { faArrowAltCircleRight, faCamera, faXmark } from "@fortawesome/free-solid-svg-icons"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import Webcam from "react-webcam"

const ChatPageContent = () => {
    const [message, setMessage] = useState('')
    const [textBox, setTextBox] = useState([])
    const [openVideo, setOpenVideo] = useState(false)

    const webRef = useRef(null)

    const handleSend = (e) => {
        e.preventDefault()
        if (message !== ''){
            setTextBox([...textBox, message])
        }
        setMessage('')
    }

    const closeVideoModal = () => {
        setOpenVideo(false)
    }

    return (
        <div className="bg-indigo-600 flex h-screen items-center justify-center relative">
            <section className="relative w-1/2 h-3/4 shadow-lg bg-white rounded-lg min-h-min">
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
                        <input value = {message} onChange={(event) => setMessage(event.target.value)} className="h-8 w-5/6 p-2 mr-4 border border-indigo-600 rounded-md" type="text"/>
                        <button onClick={handleSend} className="h-8 cursor-pointer"><FontAwesomeIcon icon={faArrowAltCircleRight} size="2x" color="#3949AB" /></button>
                    </section> 
                    <section className="mt-1 p-1  flex justify-center items-center">
                        <button className="mr-4">Process</button>
                        <FontAwesomeIcon onClick = {(event) => {setOpenVideo(!openVideo)}} className="cursor-pointer" icon={faCamera}/>
                    </section>
            </section>

            {openVideo ? 
            <div className="absolute p-8 bg-slate-400 rounded-lg">
                <FontAwesomeIcon onClick= {closeVideoModal} icon={faXmark} className="relative left-0"/>
                <Webcam ref = {webRef}/>
            </div>:
            <></>}
        </div>
    )
}
export default ChatPageContent