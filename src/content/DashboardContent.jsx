import { useState } from "react"
import { useContext } from "react"
import AuthContext from "../context/AuthContext"
import styles from "./scss/DashBoardContent.module.scss"
import classNames from "classnames/bind"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faArrowLeft, faRotateRight, faDownload } from "@fortawesome/free-solid-svg-icons"

let cx = classNames.bind(styles)

export default function DashBoardContent(){
    const {user, handleLogin} = useContext(AuthContext)
    
    const history = [
        {records_name: ""},
        {records_name: ""},
    ]

    return (
        <div className="bg-indigo-600 flex h-screen items-center justify-center relative">
            <div className={cx("modal")}>
                <h1 className={cx("dash_board_title")}>Dashboard</h1>
                <section className={cx("page_control")}>
                    <FontAwesomeIcon icon={faArrowLeft}/>
                    <FontAwesomeIcon icon={faRotateRight}/>
                </section>
                <section className={cx("")}>
                    <select name="" id="">
                    {history.map(() => {
                        
                    })}
                    </select>
                </section>
                <section>
                    <section>
                        <section>

                        </section>
                        <section>

                        </section>
                    </section>
                    <section>

                    </section>
                </section>
                <section>
                    <button><FontAwesomeIcon icon={faDownload}/>Download history</button>
                </section>
            </div>
        </div>
    )
}