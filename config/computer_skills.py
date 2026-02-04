# config/computer_skills.py

COMPUTER_SKILLS = {
    '编程语言': [
        'Python', 'Java', 'JavaScript', 'TypeScript', 'C', 'C++', 'C#', 'Go', 'Rust',
        'Kotlin', 'Swift', 'PHP', 'Ruby', 'Scala', 'R', 'MATLAB', 'SQL', 'PL/SQL',
        'Shell', 'Bash', 'PowerShell', 'Perl', 'Lua', 'Haskell', 'Elixir',
        'Dart', 'Objective-C', 'Verilog', 'VHDL', 'Solidity'
    ],
    '人工智能与大模型': [
        '机器学习', '深度学习', '自然语言处理', '计算机视觉', '强化学习',
        '大语言模型', 'LLM', 'LangChain', 'RAG', 'Prompt Engineering',
        'PyTorch', 'TensorFlow', 'Keras', 'Scikit-learn', 'Hugging Face',
        'Stable Diffusion', 'Transformer', 'BERT', 'GPT'
    ],
    '前端框架': [
        'React', 'Vue', 'Angular', 'Svelte', 'Next.js', 'Nuxt.js', 'Gatsby',
        'jQuery', 'Bootstrap', 'Tailwind CSS', 'Ant Design', 'Element UI',
        'Vuetify', 'Material UI', 'Chakra UI', 'SolidJS', 'Three.js', 'ECharts'
    ],
    '后端框架': [
        'Django', 'Flask', 'FastAPI', 'Spring', 'Spring Boot', 'Express', 'Koa',
        'NestJS', 'Laravel', 'Rails', 'ASP.NET', 'Phoenix', 'Actix', 'Rocket',
        'Sanic', 'Tornado', 'Gin', 'Fiber', 'Dubbo', 'gRPC'
    ],
    '移动开发': [
        'Flutter', 'React Native', 'Uni-app', 'Taro', 'Android SDK', 'iOS SDK',
        'HarmonyOS', '鸿蒙开发', 'WeChat Mini Program', '微信小程序'
    ],
    '大数据': [
        'Hadoop', 'Spark', 'Flink', 'Kafka', 'Hive', 'Presto', 'Impala',
        'Doris', 'StarRocks', 'Airflow', 'Zookeeper', 'DataX'
    ],
    '数据库': [
        'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch', 'Oracle',
        'SQL Server', 'SQLite', 'Cassandra', 'HBase', 'Neo4j', 'InfluxDB',
        'ClickHouse', 'DynamoDB', 'Firestore', 'Milvus', 'Pinecone', 'TiDB'
    ],
    '云平台与基础设施': [
        'AWS', 'Azure', 'Google Cloud', '阿里云', '腾讯云', '华为云', 'DigitalOcean',
        'Heroku', 'Vercel', 'Netlify', 'Cloudflare', 'Firebase', 'OpenStack', 'VMware'
    ],
    'DevOps工具': [
        'Docker', 'Kubernetes', 'K8s', 'Jenkins', 'GitLab CI', 'GitHub Actions', 'CircleCI',
        'Terraform', 'Ansible', 'Chef', 'Puppet', 'Prometheus', 'Grafana',
        'ELK Stack', 'Splunk', 'Datadog', 'ArgoCD', 'Helm'
    ],
    '版本控制': ['Git', 'SVN', 'Mercurial'],
    '测试与质量': [
        'Selenium', 'Pytest', 'JUnit', 'Cypress', 'Playwright', 'Postman',
        'JMeter', 'LoadRunner', 'SonarQube'
    ],
    '嵌入式与物联网': [
        'RTOS', 'FreeRTOS', 'STM32', 'Arduino', 'Raspberry Pi', 'ESP32',
        'MQTT', 'UART', 'SPI', 'I2C'
    ],
    '操作系统': ['Linux', 'Windows', 'macOS', 'Ubuntu', 'CentOS', 'Debian', 'Red Hat', 'Android', 'iOS'],
    '其他工具': [
        'JIRA', 'Confluence', 'Slack', 'Notion', 'Swagger', 'Figma',
        'Webpack', 'Vite', 'Babel', 'ESLint', 'Prettier', 'Jupyter', 'VS Code'
    ]
}

# 扁平化所有技能
ALL_COMPUTER_SKILLS = [skill for skills in COMPUTER_SKILLS.values() for skill in skills]

# 技能标准化映射（新增更多别名）
SKILL_NORMALIZATION = {
    # 编程语言
    'python3': 'Python', 'py3': 'Python', 'python 2': 'Python',
    'javascript': 'JavaScript', 'js': 'JavaScript',
    'typescript': 'TypeScript', 'ts': 'TypeScript',
    'golang': 'Go', 'cpp': 'C++', 'cplusplus': 'C++',
    'csharp': 'C#', 'dotnet': 'C#',

    # 前端
    'react.js': 'React', 'reactjs': 'React',
    'vue.js': 'Vue', 'vuejs': 'Vue',
    'angular.js': 'Angular', 'angularjs': 'Angular',
    'node.js': 'Node.js', 'nodejs': 'Node.js',
    'threejs': 'Three.js',

    # 移动端
    'flutter': 'Flutter', 'react-native': 'React Native', 'rn': 'React Native',

    # 后端
    'spring': 'Spring', 'springboot': 'Spring Boot', 'springmvc': 'Spring',
    'fast-api': 'FastAPI', 'django-rest-framework': 'Django',

    # 数据库
    'mysql': 'MySQL', 'postgresql': 'PostgreSQL', 'postgres': 'PostgreSQL',
    'mongodb': 'MongoDB', 'mongo': 'MongoDB',
    'redis': 'Redis', 'elasticsearch': 'Elasticsearch', 'es': 'Elasticsearch',
    'click-house': 'ClickHouse',

    # AI
    'machine learning': '机器学习', 'deep learning': '深度学习',
    'nlp': '自然语言处理', 'cv': '计算机视觉',
    'pytorch': 'PyTorch', 'tensorflow': 'TensorFlow', 'tf': 'TensorFlow',

    # DevOps
    'docker': 'Docker', 'k8s': 'Kubernetes', 'kubernetes': 'Kubernetes',
    'jenkins': 'Jenkins', 'terraform': 'Terraform', 'git-lab': 'GitLab CI',

    # 大数据
    'hadoop': 'Hadoop', 'spark': 'Spark', 'kafka': 'Kafka',
}