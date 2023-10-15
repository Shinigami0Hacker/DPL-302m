const getFileExtension = ( file_name ) => {
    let extension = file_name.split('.').pop();
    return extension.trim().toLowerCase();
}

export default function ImportChatTest( file ){
    const reader = new FileReader;
    const file_extension = getFileExtension(file.name)
    const test_data = []
    
    if (file_extension === 'csv'){


    }
    else if (file_extension === 'json'){

    }
    return test_data
}