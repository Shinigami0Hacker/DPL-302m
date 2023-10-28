import ChatPageContent from "../content/ChatPageContent"
import LoginContent from "../content/LoginContent"
import HomePageContent from "../content/HomePageContent"
import DemoNotification from "../components/notification/DemoNotification"
import SignupContent from "../content/SignupContent"
import DashBoardContent from "../content/DashboardContent"
let PublicRoutes = [
    {content: LoginContent, path: "login", onLeaveNotification: DemoNotification},
    {content: ChatPageContent, path: "chat", onLeaveNotification: DemoNotification},
    {content: HomePageContent, path: "home", onLeaveNotification: DemoNotification},
    {content: SignupContent, path: "signup", onLeaveNotification: DemoNotification},
    {content: DashBoardContent, path: "dashboard", onLeaveNotification: onLeaveNotification}
]
export default PublicRoutes