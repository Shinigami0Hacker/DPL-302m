import classNames from "classnames/bind";
import styles from "./scss/SignupContent.module.scss";
import { Fragment } from "react";
import health_care_sign_up from "../assets/HealthCareSignUp.gif";

const cx = classNames.bind(styles)

const SignupContent = () => {
    return (
        <div className="flex justify-center items-center h-screen bg-indigo-600">
            <div>
                <img className={cx("signup_logo")} src={health_care_sign_up} alt="Healthcare logo" />
                <div className={cx("")}>
                    <div><p>Smart Management System</p></div>
                    <div></div>
                    <div></div>
                    <div></div>
                </div>
            </div>
            <div className={cx("card")}>
                <h1 className={cx("signup_title")}>Sign up</h1>
                <hr />
                <form className={cx("input_form")} method="POST">
                    <label htmlFor="">Email</label>
                    <input type="text" id="email" name="email"/>
                    <section className={cx("name_input")}>
                        <div>
                            <label htmlFor="first_name">First name</label>
                            <input type="text" id="first_name" name="first_name"/>
                        </div>
                        <div>
                            <label htmlFor="last_name" >Last name</label>
                            <input type="text" id="last_name" name="last_name"/>
                        </div>
                    </section>
                        <label for="dob">Date of Birth:</label>
                        <input type="date" id="dob" name="dob"/>
                    <section>
                        <label for="dob">Gender</label>
                        <select name="" id="">
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                        <p>Are you the doctor?<input type="checkbox" id="myCheckbox"></input></p>
                    </section>
                    <button type="submit" className="block mt-5 bg-slate-600 w-full font-bold text-[#F0F8FF] rounded-lg h-10 cursor-pointer">Sign up</button>
                </form>
            </div>
        </div>
    )
}

export default SignupContent