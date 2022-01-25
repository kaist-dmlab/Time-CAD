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
        maxCount: 1,
        action: 'http://localhost:5555/upload',
        beforeUpload: file => {
            const isCSV = acceptableExts.includes(file.type)
            if (!isCSV) {
                message.error(`${file.name} is not an acceptable file!`);
            }
            return isCSV || Upload.LIST_IGNORE;
        },
        onChange(info) {
            const { status } = info.file;

            if (status === 'done') {
                const response = JSON.parse(info.file.xhr.response)
                if (response.status === 200) {
                    message.success(`${info.file.name} file uploaded successfully.`);
                    setChartDataList([...chartDataList, response.data])
                    setFileNameList([...fileNameList, info.file.name])
                } else {
                    message.error(`${info.file.name} file format is invalid.`)
                }
            } else if (status === 'error') {
                message.error(`${info.file.name} file upload failed.`);
            }
        },
    }

    React.useEffect(() => {
        setChartDataList([chartData])
        setFileNameList([fileName])
    }, [chartData, fileName]);

    return (
        <React.Fragment>
            <PageHeader
                title="Anomaly Detection Results"
                extra={[
                    <Upload {...fileProps} accept={acceptableExts} key='add-upload'><Button size='large' key="new_data" type="primary" shape='round'><PlusOutlined />New Data</Button></Upload>,
                ]} />
            <Row gutter={[16, 16]}>
                {/* TODO: Action on Add New Data */}
                {console.log(chartDataList)}
                {chartDataList.map((data, i) =>
                    <MainViewer key={i} chartData={data} fileName={fileNameList[i]} />
                )}
            </Row>
        </React.Fragment>
    )
}
