import classNames from "classnames/bind";
import styles from './scss/HomePage.module.scss'
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faGear } from "@fortawesome/free-solid-svg-icons";
import { useState } from "react";
import RoomConsoleModal from "../components/modal/RoomConsoleModal";
import { createContext } from "react";

const cx = classNames.bind(styles)

export default function HomePageContent() {
    const [ roomConsole, setRoomConsole ] = useState(false)

    let patients = [
        {id: 1, name: " Johny", isOnline: false},
        {id: 2, name: "Vinhseg", isOnline: true},
        {id: 3, name: "Tommy", isOnline: true}
    ]

    return (
        <RoomConsoleModal show={roomConsole} patients={patients} control={setRoomConsole}>
            <div className="h-screen w-screen flex justify-center items-center bg-indigo-600">
                <div className={cx("main_section")}>
                    <div className={cx("name")}>Console</div>
                    <hr className="h-1 mx-auto my-4 bg-gray-600 border-0 rounded dark:bg-gray-700"/>
                    <section className="flex flex-col" >
                        <button onClick={() => setRoomConsole(!roomConsole)} className="block bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 border border-blue-700 rounded mt-5">Create chat room</button>
                        <button className="block bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 border border-blue-700 rounded mt-10">Management</button>
                        <button className="block bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 border border-blue-700 rounded mt-10">History</button>
                    </section>
                    <section className="relative">
                        <FontAwesomeIcon icon={faGear} size="xl" className="block mt-20 absolute right-0 cursor-pointer"/>
                    </section>
                </div>
            </div>
        </RoomConsoleModal>
    )
}