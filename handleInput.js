let section = []

function pushUserInput(){

    const currentDate = new Date();

    let user_input = document.getElementById('user_input')
    let chat_section = document.getElementById('chat_section')
    let post = document.createElement("p")

    const hours = currentDate.getHours();
    const minutes = currentDate.getMinutes();
    const seconds = currentDate.getSeconds();

    if (user_input.value === ""){
        return
    }

    post.textContent = user_input.value
    post.id = "post_chat"
    
    section.push({
        chat_content: user_input.value.trim(),
        timeStamp: {
            hours: hours,
            minutes: minutes,
            seconds: seconds
        }
    })

    chat_section.appendChild(post);
    user_input.value = ""
}

function uploadToBackend(){

    if (section.length === 0){
        alert("The chat section history is empty")
        return
    }

    const httpPostContent = {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        },
        body: JSON.stringify(section)
      };

    fetch("http://127.0.0.1:5000/upload_text", httpPostContent).then((res)=>{
        let result_section = document.getElementById("result_section")
        res.json().then((res) => {
            res.forEach(result => {
                let ele = document.createElement("p")
                console.log(result[0])
                ele.textContent = `'${result[0]['chat_content']}' at ${result[0]['timeStamp']['hours']}:${result[0]['timeStamp']['minutes']}:${result[0]['timeStamp']['seconds']}- ${result[1]}`
                result_section.appendChild(ele)
            });
        })
    }).catch((err) => {
        const download_btn = doc.createElement("button")
        console.log(`Error: ${err}`)
    })

}




