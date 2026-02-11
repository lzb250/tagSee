# config/computer_skills.py

# 技能分类
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

# 技能标准化映射（别名）
SKILL_NORMALIZATION = {
    'python3': 'Python', 'py3': 'Python', 'python 2': 'Python',
    'javascript': 'JavaScript', 'js': 'JavaScript',
    'typescript': 'TypeScript', 'ts': 'TypeScript',
    'golang': 'Go', 'cpp': 'C++', 'cplusplus': 'C++',
    'csharp': 'C#', 'dotnet': 'C#',
    'react.js': 'React', 'reactjs': 'React',
    'vue.js': 'Vue', 'vuejs': 'Vue',
    'angular.js': 'Angular', 'angularjs': 'Angular',
    'node.js': 'Node.js', 'nodejs': 'Node.js',
    'threejs': 'Three.js',
    'flutter': 'Flutter', 'react-native': 'React Native', 'rn': 'React Native',
    'spring': 'Spring', 'springboot': 'Spring Boot', 'springmvc': 'Spring',
    'fast-api': 'FastAPI', 'django-rest-framework': 'Django',
    'mysql': 'MySQL', 'postgresql': 'PostgreSQL', 'postgres': 'PostgreSQL',
    'mongodb': 'MongoDB', 'mongo': 'MongoDB',
    'redis': 'Redis', 'elasticsearch': 'Elasticsearch', 'es': 'Elasticsearch',
    'click-house': 'ClickHouse',
    'machine learning': '机器学习', 'deep learning': '深度学习',
    'nlp': '自然语言处理', 'cv': '计算机视觉',
    'pytorch': 'PyTorch', 'tensorflow': 'TensorFlow', 'tf': 'TensorFlow',
    'docker': 'Docker', 'k8s': 'Kubernetes', 'kubernetes': 'Kubernetes',
    'jenkins': 'Jenkins', 'terraform': 'Terraform', 'git-lab': 'GitLab CI',
    'hadoop': 'Hadoop', 'spark': 'Spark', 'kafka': 'Kafka',
}

# 技能版本号绑定
SKILL_VERSIONS = {
    'Java': ['Java 8','Java 11','Java 17','Java 21','Java SE'],
    'Python': ['Python 2.7','Python 3.6','Python 3.7','Python 3.8','Python 3.9','Python 3.10','Python 3.11','Python 3.12'],
    'Go': ['Go 1.16','Go 1.17','Go 1.18','Go 1.19','Go 1.20','Go 1.21'],
    'Node.js': ['Node.js 10','Node.js 12','Node.js 14','Node.js 16','Node.js 18','Node.js 20'],
    'Vue': ['Vue 2','Vue 3','Vue.js 2','Vue.js 3'],
    'React': ['React 15','React 16','React 17','React 18'],
    'Angular': ['AngularJS','Angular 2','Angular 4','Angular 6','Angular 8','Angular 10','Angular 12','Angular 14','Angular 16','Angular 17'],
    'Spring': ['Spring 4','Spring 5','Spring 6','Spring Boot 1.x','Spring Boot 2.0','Spring Boot 2.1','Spring Boot 2.2','Spring Boot 2.3','Spring Boot 2.4','Spring Boot 2.5','Spring Boot 2.6','Spring Boot 2.7','Spring Boot 3.0','Spring Boot 3.1','Spring Boot 3.2','Spring Cloud 2020','Spring Cloud 2021','Spring Cloud 2022','Spring Cloud 2023'],
    'Django': ['Django 2.x','Django 3.x','Django 4.x','Django 5.x'],
    'Flask': ['Flask 1.x','Flask 2.0','Flask 2.1','Flask 2.2','Flask 2.3','Flask 3.0'],
    'MySQL': ['MySQL 5.5','MySQL 5.6','MySQL 5.7','MySQL 8.0'],
    'PostgreSQL': ['PostgreSQL 9.x','PostgreSQL 10','PostgreSQL 11','PostgreSQL 12','PostgreSQL 13','PostgreSQL 14','PostgreSQL 15','PostgreSQL 16'],
    'MongoDB': ['MongoDB 3.x','MongoDB 4.0','MongoDB 4.2','MongoDB 4.4','MongoDB 5.0','MongoDB 6.0','MongoDB 7.0'],
    'Redis': ['Redis 3.x','Redis 4.x','Redis 5.0','Redis 6.0','Redis 6.2','Redis 7.0','Redis 7.2'],
    'Elasticsearch': ['Elasticsearch 5.x','Elasticsearch 6.x','Elasticsearch 7.x','Elasticsearch 8.x'],
    'Docker': ['Docker 17.x','Docker 18.x','Docker 19.x','Docker 20.x','Docker 23.x','Docker 24.x'],
    'Kubernetes': ['Kubernetes 1.18','Kubernetes 1.19','Kubernetes 1.20','Kubernetes 1.21','Kubernetes 1.22','Kubernetes 1.23','Kubernetes 1.24','Kubernetes 1.25','Kubernetes 1.26','Kubernetes 1.27','Kubernetes 1.28','Kubernetes 1.29','K8s','K8s 1.24','K8s 1.28']
}
