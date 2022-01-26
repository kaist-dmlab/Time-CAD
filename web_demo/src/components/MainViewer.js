/* eslint-disable import/no-anonymous-default-export */
import React from 'react'
import { Card, Col, Row, Select, Checkbox } from 'antd';
import { CalendarOutlined } from '@ant-design/icons';
import Charts from './Charts'

const { Option } = Select;
const CheckboxGroup = Checkbox.Group;

export default (props) => {

    const [chartVariables, setChartVariables] = React.useState(props.chartVariables)
    const [checkedList, setCheckedList] = React.useState(props.chartVariables);
    const [currentRange, setCurrentRange] = React.useState('all');
    const [indeterminate, setIndeterminate] = React.useState(false);
    const [checkAll, setCheckAll] = React.useState(true);
    const [displayData, setDisplayData] = React.useState(props.chartData)


    React.useEffect(() => {
        setChartVariables(props.chartVariables)
        setCheckedList(props.chartVariables)
    }, [props.chartVariables]);

    const onVariableChange = list => {
        setCheckedList(list);
        setIndeterminate(!!list.length && list.length < chartVariables.length);
        setCheckAll(list.length === chartVariables.length);

        let newChartData = props.chartData.filter(data => list.includes(data.column))
        setDisplayData(newChartData)
    };

    const onCheckAllChange = e => {
        setCheckedList(e.target.checked ? chartVariables : []);
        setIndeterminate(false);
        setCheckAll(e.target.checked);

        if (e.target.checked) {
            let newChartData = props.chartData.filter(data => chartVariables.includes(data.column))
            setDisplayData(newChartData)
        } else {
            let newChartData = props.chartData.filter(data => !chartVariables.includes(data.column))
            setDisplayData(newChartData)
        }
    };

    const onRangeChange = value => {
        let newChartData = props.chartData.filter(data => chartVariables.includes(data.column))

        switch (value) {
            case '7d':
                newChartData = newChartData.slice(-7 * chartVariables.length)
                break;
            case '31d':
                newChartData = newChartData.slice(-31 * chartVariables.length)
                break;
            case '3m':
                newChartData = newChartData.slice(-90 * chartVariables.length)
                break;
            case '6m':
                newChartData = newChartData.slice(-180 * chartVariables.length)
                break;
            case '12m':
                newChartData = newChartData.slice(-356 * chartVariables.length)
                break;
            default:
                break;
        }
        setDisplayData(newChartData)
        setCurrentRange(value)
    }

    return (
        <React.Fragment>
            <Col span={4}>
                <Card title="Variable Filter" bordered={false} extra={
                    <Checkbox indeterminate={indeterminate} onChange={onCheckAllChange} checked={checkAll}>
                        Show All
                    </Checkbox>
                }>
                    <CheckboxGroup style={{ width: '100%' }} value={checkedList} onChange={onVariableChange}>
                        <Row>
                            {
                                chartVariables ? chartVariables.map((option, i) =>
                                    <Col key={i} span={24}>
                                        <Checkbox key={i} value={option}>{option}</Checkbox>
                                    </Col>
                                ) : null
                            }
                        </Row>
                    </CheckboxGroup>
                </Card>
            </Col>
            <Col span={20}>
                <Card title={"Displaying " + props.fileName} bordered={false} // TODO: click to show data table
                    extra={
                        <Select defaultValue={currentRange} style={{ width: 172, fontWeight: 'bold' }} onChange={onRangeChange}>
                            <Option key='all' value="all"><CalendarOutlined /> All</Option>
                            <Option key='7d' value="7d"><CalendarOutlined /> Last 7 days</Option>
                            <Option key='31d' value="31d"><CalendarOutlined /> Last 31 days</Option>
                            <Option key='3m' value="3m"><CalendarOutlined /> Last 3 months</Option>
                            <Option key='6m' value="6m"><CalendarOutlined /> Last 6 months</Option>
                            <Option key='12m' value="12m"><CalendarOutlined /> Last 12 months</Option>
                        </Select>
                    }>
                    <Charts chartData={displayData} chartVariables={checkedList} setCheckedList={setCheckedList} />
                </Card>
            </Col>
        </React.Fragment >
    )
}
