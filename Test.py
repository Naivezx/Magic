import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

# 測試 Keras 版本和後端
print(f"Keras 版本: {keras.__version__}")
print(f"Keras 後端: {keras.backend.backend()}")

# 創建一個簡單的序列模型來測試功能
try:
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 編譯模型
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # 打印模型摘要
    model.summary()
    
    print("\n✅ Keras 測試成功！可以正常使用。")
except Exception as e:
    print(f"\n❌ Keras 測試失敗！錯誤信息：\n{str(e)}")