/* eslint-disable import/no-anonymous-default-export */
import React from 'react'
import { Row, PageHeader, Button, Upload } from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import MainViewer from './MainViewer'

export default ({ chartData }) => {

    const [chartDataList, setChartDataList] = React.useState([])

    React.useEffect(() => {
        setChartDataList([chartData])
    }, [chartData]);

    const onAddData = value => {

    }

    return (
        <React.Fragment>
            <PageHeader
                title="Anomaly Detection Results"
                extra={[
                    <Upload key='add-upload'><Button size='large' key="new_data" type="primary" shape='round' onClick={onAddData}><PlusOutlined />New Data</Button></Upload>,
                ]} />
            <Row gutter={[16, 16]}>
                {/* TODO: Loop to create MainViewer over each chartData in chartDataList + Action on Add New Data */}
                {chartDataList.map((data, i) =>
                    <MainViewer key={i} chartData={data} />
                )}
            </Row>
        </React.Fragment>
    )
}
