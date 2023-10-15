import { useContext } from "react"
import { Route } from "react-router-dom"
import { Navigate } from "react-router-dom"

const PrivateRoute = ({ children, ...route_setting}) => {
    let {user} = useContext(AuthContext)
    return (
        <Route {...route_setting}>
        {!user ? <Navigate to={"login/"} replace={true}/> : children}</Route>
    )
}
export default PrivateRoute