from setuptools import setup, find_packages

setup(
    name="foundationpose",
    version="0.1.0",
    description="该模型可以根据物品mask图和CAD的3D模板来对物品进行位姿估计。它的输入是一张RGB图像和一张深度图、一张mask、CAD模板以及相机的内参，输出的是物品的姿态。 它不需要训练，使用提供的权重即可。 有了它，可以在实际对物品抓取如新零售等场景进行落地。",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="http://192.168.0.188:8090/ai_lab_rd02/ai_sdks/foundationpose.git",
    # author="sam", # 作者
    # author_email="<EMAIL>", # 作者邮箱
    # license="MIT", # 许可证
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        "ultralytics~=8.0.120",
        "opencv-python~=4.10.0.84"
    ],
    include_package_data=True,
    zip_safe=False,
)
