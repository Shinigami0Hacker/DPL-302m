import { Button } from "react-bootstrap"
import { Modal } from "react-bootstrap"
import styles from "./scss/RoomConsole.Modal.module.scss"
import classNames from "classnames/bind";

let cx = classNames.bind(styles)

function RoomConsoleModal({ show, children }) {
    return (
      show ? <></> :
      <div className="relative">
      <div className={"absolute w-screen h-screen flex " + cx("modal_background")}>
      </div>
      { children }
      </div>
    );
}  
export default RoomConsoleModal;
