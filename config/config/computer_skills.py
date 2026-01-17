# config/computer_skills.py
COMPUTER_SKILLS = {
    '编程语言': [
        'Python', 'Java', 'JavaScript', 'TypeScript', 'C', 'C++', 'C#', 'Go', 'Rust',
        'Kotlin', 'Swift', 'PHP', 'Ruby', 'Scala', 'R', 'MATLAB', 'SQL', 'PL/SQL',
        'Shell', 'Bash', 'PowerShell', 'Perl', 'Lua', 'Haskell', 'Elixir'
    ],
    '前端框架': [
        'React', 'Vue', 'Angular', 'Svelte', 'Next.js', 'Nuxt.js', 'Gatsby',
        'jQuery', 'Bootstrap', 'Tailwind CSS', 'Ant Design', 'Element UI',
        'Vuetify', 'Material UI', 'Chakra UI'
    ],
    '后端框架': [
        'Django', 'Flask', 'FastAPI', 'Spring', 'Spring Boot', 'Express', 'Koa',
        'NestJS', 'Laravel', 'Rails', 'ASP.NET', 'Phoenix', 'Actix', 'Rocket'
    ],
    '数据库': [
        'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch', 'Oracle',
        'SQL Server', 'SQLite', 'Cassandra', 'HBase', 'Neo4j', 'InfluxDB',
        'ClickHouse', 'DynamoDB', 'Firestore'
    ],
    '云平台': [
        'AWS', 'Azure', 'Google Cloud', '阿里云', '腾讯云', '华为云', 'DigitalOcean',
        'Heroku', 'Vercel', 'Netlify', 'Cloudflare', 'Firebase'
    ],
    'DevOps工具': [
        'Docker', 'Kubernetes', 'Jenkins', 'GitLab CI', 'GitHub Actions', 'CircleCI',
        'Terraform', 'Ansible', 'Chef', 'Puppet', 'Prometheus', 'Grafana',
        'ELK Stack', 'Splunk', 'Datadog'
    ],
    '版本控制': ['Git', 'SVN', 'Mercurial'],
    '操作系统': ['Linux', 'Windows', 'macOS', 'Ubuntu', 'CentOS', 'Debian', 'Red Hat'],
    '算法与AI': [
        '机器学习', '深度学习', '计算机视觉', '自然语言处理', '强化学习',
        '数据挖掘', '统计分析', '时间序列分析'
    ],
    '其他工具': [
        'JIRA', 'Confluence', 'Slack', 'Notion', 'Postman', 'Swagger', 'Figma',
        'Webpack', 'Vite', 'Babel', 'ESLint', 'Prettier', 'Jupyter', 'VS Code'
    ]
}

# 扁平化所有技能
ALL_COMPUTER_SKILLS = [skill for skills in COMPUTER_SKILLS.values() for skill in skills]

# 技能标准化映射
SKILL_NORMALIZATION = {
    # 编程语言
    'python3': 'Python', 'py3': 'Python', 'python 3': 'Python',
    'javascript': 'JavaScript', 'js': 'JavaScript',
    'typescript': 'TypeScript', 'ts': 'TypeScript',
    'golang': 'Go', 'golang': 'Go',

    # 前端
    'react.js': 'React', 'reactjs': 'React',
    'vue.js': 'Vue', 'vuejs': 'Vue',
    'angular.js': 'Angular', 'angularjs': 'Angular',
    'node.js': 'Node.js', 'nodejs': 'Node.js',

    # 后端
    'express.js': 'Express', 'expressjs': 'Express',
    'spring boot': 'Spring Boot',

    # 数据库
    'mysql': 'MySQL', 'postgresql': 'PostgreSQL', 'postgres': 'PostgreSQL',
    'mongodb': 'MongoDB', 'mongo': 'MongoDB',
    'redis': 'Redis', 'elasticsearch': 'Elasticsearch', 'es': 'Elasticsearch',

    # DevOps
    'docker': 'Docker', 'k8s': 'Kubernetes', 'kubernetes': 'Kubernetes',
    'jenkins': 'Jenkins', 'terraform': 'Terraform',

    # 版本控制
    'git': 'Git', 'svn': 'SVN',

    # 操作系统
    'linux': 'Linux', 'ubuntu': 'Ubuntu', 'centos': 'CentOS', 'debian': 'Debian',
    'windows': 'Windows', 'macos': 'macOS'
}