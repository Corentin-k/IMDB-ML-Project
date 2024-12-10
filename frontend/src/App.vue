<template>
  <body>
  <div id="app">
    <SentimentAnalysis />
  </div>
  </body>
</template>

<script>
import SentimentAnalysis from "./components/SentimentAnalysis.vue";

export default {
  name: "App",
  components: {
    SentimentAnalysis,
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  text-align: center;
  margin: 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 90vh;

}

body {
  background-color: #302e2e;;
}


</style>
