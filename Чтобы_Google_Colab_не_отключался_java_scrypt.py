# Обсуждение
https://stackoverflow.com/questions/57113226/how-to-prevent-google-colab-from-disconnecting

#YouTube демонстрация rкуда вставлять скрипт!
https://www.youtube.com/watch?v=RpQaAbMmqkA

# функция которую вставляем!
var startClickConnect = function startClickConnect(){
var clickConnect = function clickConnect(){
console.log("Connnect Clicked - Working");
document.querySelector("[icon='colab:folder-refresh']").click()
};
 
var intervalId = setInterval(clickConnect, 60000);
 
var stopClickConnectHandler = function stopClickConnect() {
clearInterval(intervalId);
console.log("Connnect Clicked Stopped");
};
 
return stopClickConnectHandler;
};
 
#запуск непрерывного кликанья по кнопке обновить ресурсы =)
var stopClickConnect = startClickConnect();
 
#остановка непрерывного кликанья по кнопке обновить ресурсы =)
stopClickConnect();