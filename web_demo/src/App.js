import { Layout, Menu, Breadcrumb } from 'antd';
import './App.css';

const { Header, Content, Footer } = Layout;



function App() {
  return (
    <Layout className="layout">
      <Header style={{ "backgroundColor": "white" }}>
        <Menu theme="light" mode="horizontal">
          <Menu.Item key="home">
            <a href="/">
              Time-CAD
            </a>
          </Menu.Item>
        </Menu>
      </Header>
      <Content style={{ padding: '0 50px' }}>
        <br />
        <div className="site-layout-content">
          Content
        </div>
      </Content>
      <Footer style={{ textAlign: 'center' }}><em>Time-CAD</em> © 2022 Created by <a href="https://dm.kaist.ac.kr">KAIST Data Mining Lab</a></Footer>
    </Layout>
  );
}

export default App;