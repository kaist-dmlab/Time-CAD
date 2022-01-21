/* eslint-disable import/no-anonymous-default-export */
import React from 'react'
import { Card, Col, Row, PageHeader, Button, Divider, Select, Checkbox, message } from 'antd';
import { PlusOutlined, CalendarOutlined } from '@ant-design/icons';
import Charts from './Charts'

const { Option } = Select;
const CheckboxGroup = Checkbox.Group;

function handleChange(value) {
    message.info(`selected ${value}`);
}

export default props => {
    let plainOptions = ['Column A', 'Column B', 'Column C'];
    let current_range = 'all'

    const [checkedList, setCheckedList] = React.useState(plainOptions);
    const [indeterminate, setIndeterminate] = React.useState(true);
    const [checkAll, setCheckAll] = React.useState(false);

    const onChange = list => {
        setCheckedList(list);
        setIndeterminate(!!list.length && list.length < plainOptions.length);
        setCheckAll(list.length === plainOptions.length);
    };

    const onCheckAllChange = e => {
        setCheckedList(e.target.checked ? plainOptions : []);
        setIndeterminate(false);
        setCheckAll(e.target.checked);
    };

    return (
        <React.Fragment>
            <PageHeader
                title="Anomaly Detection Results"
                extra={[
                    <Button size='large' key="new_data" type="primary" shape='round'><PlusOutlined />New Data</Button>,
                ]} />
            <Row gutter={16}>
                <Col span={4}>
                    <Card bordered={false}>
                        <a href="#">filename</a>
                        <Divider orientation='left'>Variable Filter</Divider>
                        <Checkbox indeterminate={indeterminate} onChange={onCheckAllChange} checked={checkAll}>
                            Check all
                        </Checkbox>
                        <CheckboxGroup style={{ width: '100%' }} value={checkedList} onChange={onChange}>
                            <Row>
                                {
                                    plainOptions.map(option =>
                                        <Col span={24}>
                                            <Checkbox value={option}>{option}</Checkbox>
                                        </Col>
                                    )
                                }
                            </Row>
                        </CheckboxGroup>
                    </Card>
                </Col>
                <Col span={20}>
                    <Card bordered={false}
                        extra={
                            <Select defaultValue={current_range} style={{ width: 172, fontWeight: 'bold' }} onChange={handleChange}>
                                <Option value="all"><CalendarOutlined /> All</Option>
                                <Option value="7d"><CalendarOutlined /> Last 7 days</Option>
                                <Option value="31d"><CalendarOutlined /> Last 31 days</Option>
                                <Option value="3m"><CalendarOutlined /> Last 3 months</Option>
                                <Option value="6m"><CalendarOutlined /> Last 6 months</Option>
                                <Option value="12m"><CalendarOutlined /> Last 12 months</Option>
                            </Select>
                        }>
                        <Charts />
                    </Card>
                </Col>
            </Row>
        </React.Fragment>
    )
}
