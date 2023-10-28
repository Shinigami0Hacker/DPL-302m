import styles from "./scss/RoomConsole.Modal.module.scss"
import classNames from "classnames/bind";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faCircleXmark } from "@fortawesome/free-solid-svg-icons";
import { useNavigate } from "react-router-dom";
import { Link } from "react-router-dom";

let cx = classNames.bind(styles)

function RoomConsoleModal({ patients, children, show, control}) {
    const navigate = useNavigate();
    return (
      show ?
      <div className="relative">
        <div className={"absolute w-screen h-screen flex " + cx("modal_background")}>
          <div className={cx("room_console_modal")}>
              <button onClick={() => control(!show)} className="absolute right-5 top-5">
                <FontAwesomeIcon icon={faCircleXmark} size="xl" color="#eb4034">
              </FontAwesomeIcon>
              </button>
              <p className={cx("room_console_title")}>Room selection</p>
              <a href="/chat"></a>
              <form className={cx("room_console_form")} action="">
                <select className={cx("room_console_selection")}>
                  {patients.map((patient, index) => {
                    return (
                      <option value={patient.id} key={index}>
                          {patient.name}
                      </option>
                    )
                  })}
                </select>
                <div className={cx("btn_wrapper")}>
                  <button onClick={() => navigate("/chat")} className= {"block bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 border border-blue-700 rounded mt-10 " + cx("room_console_btn")}>Create room</button>
                </div>
              </form>
          </div>
      </div>
      { children }
      </div> : <>{children}</>
    );
}  
export default RoomConsoleModal;
