window.onload = function() {
    var oTab = document.getElementById("tabs");
    var oU1 = oTab.getElementsByTagName("ul")[0];
    var oLis = oU1.getElementsByTagName("li");
    var oDivs = oTab.getElementsByTagName('div');
    for (var i=0, len = oLis.length; i<len; i++){
        oLis[i].index = i;
        oLis[i].onclick = function(){
            for (var n=0; n<len; n++){
                oLis[n].className = "";
                oDivs[n].className = "hide";
            }
            this.className = "on";
            oDivs[this.index].className = "";
        }
    }
}         