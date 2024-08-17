# Vue基础语法

## 基础举例

'''
import { createApp, ref } from 'vue'

createApp({
    <!--createApp 创建Vue应用实例-->
  setup() {
    <!--setup为自定义函数名-->
    <!--setup函数返回count的值-->
    return {
        <!--ref赋值-->
        <!--响应式??-->
      count: ref(0)
    }
  }
}).mount('#app')
<!--使用.mount()方法将应用挂载到id为app的元素上-->
'''

> Vue .mount()
> `https://blog.csdn.net/qq_23730073/article/details/135828687`
