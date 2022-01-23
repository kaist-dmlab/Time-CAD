/* eslint-disable import/no-anonymous-default-export */
import React from 'react'
import { Row, PageHeader, Button, Upload, message } from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import MainViewer from './MainViewer'

import data2 from '../chart_data/data2.json'

// Later screen shown after the user uploads a file.
// New data uploaded through the add data button
// More charts shown as additional files are uploaded
// Columns shown and hidden with the toggle buttons on the left
export default ({ chartData, fileName }) => {

    const acceptableExts = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
    const [chartDataList, setChartDataList] = React.useState([])
    const [fileNameList, setFileNameList] = React.useState([])

    const fileProps = {
        multiple: false,
        beforeUpload: file => {
            const isOkay = acceptableExts.includes(file.type)
            if (!isOkay) {
                message.error(`${file.name} is not a png file`);
            }
            return isOkay || Upload.LIST_IGNORE;
        },
        onChange: info => {
            console.log(info.fileList);
        },
    }

    React.useEffect(() => {
        setChartDataList([chartData])
        setFileNameList([fileName])
    }, [chartData, fileName]);

    const onAddData = value => {
        setChartDataList([...chartDataList, data2])
        setFileNameList([...fileNameList, "data2.json"])
    }

    return (
        <React.Fragment>
            <PageHeader
                title="Anomaly Detection Results"
                extra={[
                    <Upload {...fileProps} accept={acceptableExts} key='add-upload'><Button size='large' key="new_data" type="primary" shape='round' onClick={onAddData}><PlusOutlined />New Data</Button></Upload>,
                ]} />
            <Row gutter={[16, 16]}>
                {/* TODO: Loop to create MainViewer over each chartData in chartDataList + Action on Add New Data */}
                {console.log(chartDataList)}
                {chartDataList.map((data, i) =>
                    <MainViewer key={i} chartData={data} fileName={fileNameList[i]} />
                )}
            </Row>
        </React.Fragment>
    )
}
