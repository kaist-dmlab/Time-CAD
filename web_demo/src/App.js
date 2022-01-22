import React from 'react'
import { Layout } from 'antd';
import ViewWrapper from './components/ViewWrapper'
import Uploader from './components/Uploader'
import './App.css';
import data1 from './chart_data/data1.json'

const { Header, Content, Footer } = Layout;

// use local storage to save user's uploaded files or file path or use read file system for local file
// use local server to upload and then load back as json, i.e., allow both .csv .xlsx and .json ?
function App() {

  const [upload, setUpload] = React.useState(false);
  const [chartData, setChartData] = React.useState([]);

  // Similar to componentDidMount and componentDidUpdate:
  React.useEffect(() => {
    const uploadedFiles = localStorage.getItem('uploadedFiles')
    if (uploadedFiles) {
      setUpload(true)
      setChartData(data1)
    }
  }, []);

  return (
    <Layout className="layout">
      <Header style={{ "backgroundColor": "white" }}>
        <a href="/" className='menu-url'><div className='logo'></div>Time-CAD</a><span style={{verticalAlign: 'top'}}> Time-Series Anomaly Detection with Context-Aware Decomposition</span>
      </Header>
      <Content style={{ padding: '0 50px' }}>
        <br />
        <div className="site-layout-content">
          {
            upload ? <ViewWrapper chartData={chartData} setChartData={setChartData}/> : <Uploader setUpload={setUpload} setChartData={setChartData}/>
          }
        </div>
      </Content>
      <Footer style={{ textAlign: 'center' }}><em>Time-CAD</em> © 2022 Developed by <a target='_blank' rel="noreferrer" href="https://dm.kaist.ac.kr">KAIST Data Mining Lab</a></Footer>
    </Layout>
  );
}

export default App;