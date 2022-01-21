import { Layout } from 'antd';
import MainViewer from './components/MainViewer'
import Uploader from './components/Uploader'
import './App.css';

const { Header, Content, Footer } = Layout;
let upload = false

function App() {
  return (
    <Layout className="layout">
      <Header style={{ "backgroundColor": "white" }}>
        <a href="/" className='menu-url'><div className='logo'></div>Time-CAD</a>
      </Header>
      <Content style={{ padding: '0 50px' }}>
        <br />
        <div className="site-layout-content">
          {
            upload ? <MainViewer /> : <Uploader />
          }
        </div>
      </Content>
      <Footer style={{ textAlign: 'center' }}><em>Time-CAD</em> © 2022 Created by <a target='_blank' href="https://dm.kaist.ac.kr">KAIST Data Mining Lab</a></Footer>
    </Layout>
  );
}

export default App;