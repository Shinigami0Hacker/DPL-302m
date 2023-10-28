import { useCallback, useEffect, useRef, useState } from "react"
import { faArrowAltCircleRight, faCamera, faXmark } from "@fortawesome/free-solid-svg-icons"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import Webcam from "react-webcam"
import styles from "./scss/ChatPageContent.module.scss"
import classNames from "classnames/bind"
import TextSocketContext from "../context/TextSocketContex"
import AuthContext from "../context/AuthContext"
import { useContext } from "react"

const cx = classNames.bind(styles)

const ChatPageContent = () => {
    
    // const [ user ] = useContext(AuthContext)
    // const [] = useContext(TextSocketContext)

    const [message, setMessage] = useState('')
    const [textBox, setTextBox] = useState([])
    const [openVideo, setOpenVideo] = useState(false)
    const webRef = useRef(null)

    // window.addEventListener("beforeunload", () => {
    //     console.log("You out")
    // });

    const handleSend = (e) => {
        e.preventDefault()
        // textChatSocket.send(JSON.stringify({
        //     'message':message
        // }))
        const date_obj = new Date;
            let time_string = '12-30-203'
            setTextBox([...textBox, {message: message, role: " ", time_stamp: time_string}])
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
            <section className="transform ease-linear relative w-1/2 shadow-lg bg-white rounded-lg min-h-min">
                    <p className={cx("recipient")}>{"John"}</p>
                    <section className={"m-auto mt-2 overflow-y-auto w-11/12 border border-indigo-600 rounded-lg " + cx("chat_section")}>
                        {textBox.map((box, index) => {
                            return (
                                <div key = {index} className=" flex justify-end">
                                    <div className="bg-slate-400 p-2 mt-4 mr-4 rounded-md">
                                        {box.message}
                                    </div>
                                </div>
                            )})}
                    </section>
                    <section className="flex mt-5 items-center justify-center">
                        <input value = {message} onChange={(event) => setMessage(event.target.value)} className="h-8 w-5/6 p-2 mr-4 border border-indigo-600 rounded-md hover:none" type="text"/>
                        <button onClick={handleSend} className="h-8 cursor-pointer"><FontAwesomeIcon icon={faArrowAltCircleRight} size="2x" color="#3949AB" /></button>
                    </section> 
                    <section className="mt-1 p-1  flex justify-center items-center">
                        <button className="bg-red-500 hover:bg-red-400 text-white font-bold py-2 px-4 border-b-4 border-red-700 hover:border-red-500 rounded">Process</button>
                        <button className="mr-4 ml-4"><FontAwesomeIcon size="2x"onClick = {(event) => {setOpenVideo(!openVideo)}} className="cursor-pointer" icon={faCamera}/></button>
                        <button className="bg-blue-500 hover:bg-blue-400 text-white font-bold py-2 px-4 border-b-4 border-blue-700 hover:border-blue-500 rounded">Import</button>
                    </section>
            </section>
        </div>
    )
}
export default ChatPageContent