import ChatPageContent from "../content/ChatPageContent"
import LoginContent from "../content/LoginContent"
import HomePageContent from "../content/HomePageContent"

let PublicRoutes = [
    {content: LoginContent, path: "login"},
    {content: ChatPageContent, path: "chat"},
    {content: HomePageContent, path: "home"},
]
export default PublicRoutes