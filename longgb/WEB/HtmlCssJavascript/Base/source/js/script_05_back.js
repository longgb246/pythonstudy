function checkall(){
    var hobby = document.getElementsByTagName("input");
    for (var i=0; i<hobby.length; i++){
        if (hobby[i].type == 'checkbox'){
            hobby[i].checked = true;   
        }
    }
}
function clearall(){
    var hobby = document.getElementsByName("hobby");
    for (var i=0; i<hobby.length; i++){
        hobby[i].checked = false;   
    }
}

function checkone(){
    var hobby = document.getElementsByName("hobby");
    var j=parseInt(document.getElementById("wb").value);
    for (var i=0; i<hobby.length; i++){
        if ((i+1)==j){
            hobby[i].checked = true;       
        }
        else{
            hobby[i].checked = false;         
        }
    }
}


// 用于说明script的载入顺序影响了html结果
// document.write('<hr/>hobby.length<br/>');
// var hobby_1 = document.getElementsByTagName("input");
// document.write(hobby_1.length+'<br/>');
// document.write(hobby_1[1].type+'<br/>');
// document.write(hobby_1[1].getAttribute('type'));
    