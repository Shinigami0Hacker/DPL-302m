import ChatPageContent from "../content/ChatPageContent";
import { useContext } from "react";
import { TextSocketProvider } from "../context/TextSocketContex";

export default function ChatPageWrapper({ children }){
    return (
    <TextSocketProvider>
        { children }
    </TextSocketProvider>)
}