pushUserInput = () => {

    let user_input = document.getElementById('user_input')
    let chat_section = document.getElementById('chat_section')
    let post = document.createElement("p")
    
    if (user_input.value === ""){
        return
    }

    post.textContent = user_input.value
    post.id = "post_chat"

    chat_section.appendChild(post);
    user_input.value = ""
}

processModel = () => {
    console.log("Hello world")
}