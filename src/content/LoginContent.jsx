import { useEffect, useState } from "react"
import React from "react"
import AuthContext from "../context/AuthContext"

const LoginContent = () => {    
    let {user, handleLogin} = React.useContext(AuthContext)

    let [username, setUsername] = useState()
    let [password, setPassword] = useState()
    
    const handleSubmit = (e) => {
        e.preventDefault()
    }
    
    return (
        <div className="flex justify-center items-center h-screen bg-indigo-600">
            <div className="w-96 p-6 shadow-lg bg-white rounded-lg min-h-min">
                <h1 className=" text-2xl font-bold text-center font-mono">Login</h1>
                <hr className="mt-3 w-f h-1 mx-auto my-4 bg-gray-600 border-0 rounded  dark:bg-gray-700"/>
                <form action="">
                    <div className="">
                        <label htmlFor="username" className="block mt-4 text-lg">Email</label>
                        <input autoComplete="off" tabIndex="1" type="text" name="username" id="username" className="mt-1 w-full h-8 p-2 border border-indigo-600"/> <br />
                        <label htmlFor="password" className="block mt-4 text-lg">Password</label>
                        <input tabIndex="1" type="password" name="password" id="password" className="mt-1 w-full h-8 p-2 border border-indigo-600"/> <br />
                        <section className="flex mt-4">
                            <section className="basis-1/2">
                                <label htmlFor="is_doctor">Are you a doctor?</label>
                                <input type="checkbox" name= "is_doctor" className="ml-1"/>
                            </section>
                            <a className="block basis-1/2" href="">Forgot your password?</a>
                        </section>
                        <button type="submit" onClick={handleLogin} className="block mt-5 bg-slate-600 w-full font-bold text-[#F0F8FF] rounded-lg h-10 cursor-pointer">Login</button>
                        <p className="text-center mt-4 text-xs">Version 1.0</p>
                    </div>
                </form>
            </div>
        </div>
    )
}
export default LoginContent